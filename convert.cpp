#include <openvdb/openvdb.h>
#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/Interpolation.h>
#include <iostream>
#include <fstream>

int main() {
    openvdb::initialize();
    openvdb::io::File file("/home/wilhem/Downloads/smoke.vdb");
    file.open();

    openvdb::GridBase::Ptr baseGrid = file.readGrid("density");
    file.close();
    openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

    assert(grid->transformPtr()->hasUniformScale());
    std::cout  << grid->transformPtr()->voxelSize().x() << std::endl;
    auto accessor = grid->getConstAccessor();

    std::ofstream out("out.dat");


    openvdb::CoordBBox box;
    grid->tree().evalActiveVoxelBoundingBox(box);
    unsigned int s;
    s = box.min().x();
    out.write((const char*)&s, 4);
    s = box.min().y();
    out.write((const char*)&s, 4);
    s = box.min().z();
    out.write((const char*)&s, 4);
    s = box.max().z();
    out.write((const char*)&s, 4);
    s = box.max().z();
    out.write((const char*)&s, 4);
    s = box.max().z();
    out.write((const char*)&s, 4);

    for (unsigned int z = box.min().z(); z <= box.max().z(); z++)
    {
        for (unsigned int y = box.min().y(); y <= box.max().y(); y++)
        {
            for (unsigned int x = box.min().x(); x <= box.max().x(); x++)
            {
                openvdb::Vec3R p((float)x, (float)y, (float)z);
                float val = openvdb::tools::PointSampler::sample(accessor, p);
                out.write((const char*)&val, 4);
            }
        }
    }
    out.close();
    return 0;
}
